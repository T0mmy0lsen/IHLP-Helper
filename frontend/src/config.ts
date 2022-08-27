import {Main, Button, Action, Typography, Route, Sider, Section, Get} from 'react-antd-admin-panel';
import {faSearch} from "@fortawesome/free-solid-svg-icons";
import Error from "./views/Error";
import Search from "./views/Search";

let isProd = process.env.NODE_ENV === 'production';

export default {
    config: {
        debug: isProd ? 0 : 1,
        drawer: { collapsed: false },
        pathToApi: 'http://127.0.0.1:8000/ihlp',
        pathToLogo: { src: '/logo192.png', height: 30, width: 30 },
        fallbackApi: 'http://localhost',
        fallbackApiOn: [404],
        defaultRoute: () => '/',
        profile: (next: any) => {
            next(new Section()
                .addRowEnd([
                    new Typography()
                        .style({ marginTop: 1 })
                        .label(`IHLP Helper`)
                ])
            );
        },
        access: {
            accessViolationRoute: (main: Main) => {
                main.$route(`/error`)
            },
            accessViolationApi: (main: Main) => {
                main.tsxErrorMessage('Du har ikke adgang til denne funktionalitet');
            },
            access: () =>
            {
                return ({ hidden: false, access: true });
            },
        },
        bootFailed: () => {
            window.location.href = '/';
        },
        boot: (main: Main, next: any) => {
            next()
        }
    },
    routes: [
        new Route().key('/')
            .exact()
            .component(Search),
        new Route().key('/error')
            .exact()
            .component(Error),
    ],
    drawer: (next: any) => {
        next(new Section().add(new Sider()
            .add(new Button().access(true)
                .action(new Action().route(() => '/').label('Search').fontawesome(faSearch)))
        ));
    },
}