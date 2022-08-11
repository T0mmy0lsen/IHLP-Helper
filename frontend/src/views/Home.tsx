import React from "react";
import {
    Typography,
    ListHeader,
    Formula,
    Section,
    Button,
    Action,
    Cycle,
    Space,
    Title,
    List,
    Post,
    Main,
    Get
} from 'react-antd-admin-panel';
import {SectionComponent} from 'react-antd-admin-panel';
import {WarningOutlined} from "@ant-design/icons/lib";
import {message} from "antd";
import {Autocomplete, Conditions, ConditionsItem, Item} from "react-antd-admin-panel/dist";

export default class Home extends React.Component<any, any> {

    constructor(props) {
        super(props);
        this.state = {
            section: false,
            id: false,
        }
    }

    build() {

        const search = new Autocomplete();
        const button = new Button();
        const section = new Section();

        let condition = new Conditions()
            .default(() => ({ value: undefined, loading: false }))
            .add(new ConditionsItem()
                .condition((v: any) => {
                    return !!v.value
                })
                .content((next, callback, main, args) => {
                    next(new Section()
                        .add(new Space().top(24))
                        .add(new List()
                            .unique((v) => v.id)
                            .fetch(() => new Get()
                                .target(() => ({
                                    target: `/result/${args.value}`,
                                })))
                            .header(false)
                            .footer(false)
                            .headerCreate(false)
                            .headerPrepend(new ListHeader().key('id').title('ID').searchable())
                            .headerPrepend(new ListHeader().key('value').title('Name').searchable())
                            .actions(new Action()
                                .key('deleteConfirm')
                                .formula(new Formula(new Post()
                                    .target(() => ({
                                        method: 'POST',
                                        target: `/result`
                                    }))
                                    .onThen(() => {
                                        message.success('The item was deleted.')
                                    })
                                    .onCatch(() => { message.error('The item was not deleted.') })
                                )))
                        )
                    )
                })
            )
            .add(new ConditionsItem()
                .condition((v: any) => {
                    return !v.value
                })
                .content((next, callback, main, args) => next(new Section()
                    .add(new Space().top(24))
                    .add(new Typography().label('Find a request to see the result'))
                ))
            );

        button
            .action(new Action()
                .type('callback')
                .label('Clear')
                .callback(() => {
                    search.tsxSetDisabled(false);
                })
            );

        search.key('requestId')
            .label('Find a request')
            .clearable(false)
            .get((value) => new Get()
                .target(() => ({
                    target: `/search/${value}`,
                }))
                .alter((v: any) => v
                    .map((r: any) => new Item(r.id).id(r.id).title(r.value).text(r.value).object(r))
                )
            )
            .onChange((object: any) => {
                if (object.value) {
                    search.tsxSetDisabled(true);
                    condition.checkCondition({ value: object.value, loading: false })
                }
            });

        section.style({ padding: '24px 36px' });
        section.add(new Title().label('Search for a Request').level(1));
        section.add(new Space().top(12));
        section.addRowEnd([button]);
        section.add(new Space().top(12));
        section.add(search);
        section.add(condition);

        this.setState({ section: section });
    }

    render() {
        return (
            <>{!!this.state.section &&
            <SectionComponent key={this.state.id} main={this.props.main} section={this.state.section}/>}</>
        );
    }

    componentDidMount() {
        this.build()
    }
}